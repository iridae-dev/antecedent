//! The one fit-and-assemble implementation of a `TemporalDag` temporal effect.
//!
//! The sealed operation ([`super::CheckedTemporalEffectExecution`]) and the
//! one-shot route (`Study::execute_temporal`) both resolve a proof, gate their
//! own licence, and call [`fit_temporal_effect`]; the estimator, the refuters,
//! the diagnostics and the provenance names live here only. Callers own what
//! they admit (the sealed operation admits no split, custom validator or
//! Bayesian fit, the one-shot route admits them), never how an admitted fit is
//! computed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::sequential_validation::SequentialValidation;
use super::{
    Arc, BayesianConfig, BayesianGCompWorkspace, BayesianGComputationAte, BayesianTemporalGcomp,
    CausalError, CausalPosterior, Diagnostic, DiagnosticKind, DiagnosticSeverity, EffectEstimate,
    EstimationWorkspace, EstimatorId, ExecutionContext, IdentificationResult, IdentifiedEstimand,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId, InferenceMode, OverlapPolicy,
    PhysicalExecutionPlan, PosteriorPredictiveCheck, PredictiveCheckReport,
    PreparedBayesianProblem, PriorPredictiveCheck, RefutationRun, RefuteSuite,
    SUSTAINED_LINEAR_SEM, Strategy, StudyResult, TabularData, TemporalDag, TemporalEffectQuery,
    TemporalIndexer, TemporalLinearAdjustment, TemporalRefitContext, TimeSeriesData,
    bayesian_gcomp, bayesian_temporal_gcomp, effect_from_posterior,
    evaluate_bayesian_prior_sensitivity, provenance_ids, push_conflict_diagnostics,
    require_identified, resolve_bayesian_prior_with_conflict, sequential_linear_sem_assumption,
    temporal_refuter_average, with_conflict_summary, with_prior_sensitivity,
};

/// Everything one temporal-effect fit reads, resolved by the caller.
pub(super) struct TemporalEffectInputs<'a> {
    pub data: &'a TimeSeriesData,
    pub graph: &'a TemporalDag,
    pub query: &'a TemporalEffectQuery,
    pub identification: &'a IdentificationResult,
    pub estimand: &'a IdentifiedEstimand,
    pub indexer: &'a TemporalIndexer,
    pub inference: &'a InferenceMode,
    /// A sealed Bayesian proof, whose retained fit replaces the builder-config
    /// prior resolution.
    pub checked_bayesian: Option<&'a crate::analysis::CheckedBayesianTemporalEffectOperation>,
    pub bootstrap_replicates: u32,
    pub refute: RefuteSuite,
    pub custom_validators: &'a [Arc<dyn antecedent_validate::CustomEffectValidator>],
    pub split: Option<&'a antecedent_data::DiscoveryEstimationSplit>,
    pub latency_mode: Option<crate::analysis::latency::LatencyMode>,
    /// Whether the graph came from completing a PAG to a DAG; the result says so.
    pub completed_to_dag: bool,
}

/// An assembled temporal-effect fit, before the result envelope.
pub(super) struct TemporalEffectFit {
    pub estimate: EffectEstimate,
    pub estimator_id: EstimatorId,
    pub refutations: Vec<antecedent_validate::RefutationReport>,
    pub diagnostics: Vec<Diagnostic>,
    pub posterior: Option<CausalPosterior>,
    pub predictive_checks: Vec<PredictiveCheckReport>,
    pub bootstrap_replicates_ok: Option<u32>,
    pub cancelled: bool,
    /// A point value with no published interval or posterior.
    pub point_only: bool,
    estimate_provenance: (&'static str, &'static str),
}

/// How the fit's identification was obtained and how long the fit took.
#[derive(Clone, Copy)]
pub(super) struct FitFinishMeta {
    pub(super) identify_cached: bool,
    pub(super) structure_version: u32,
    pub(super) wall_time_ns: u64,
}

impl TemporalEffectFit {
    /// The result-assembly arguments of this fit on the proof it ran against.
    pub(super) fn finish_args<'a>(
        self,
        physical: &'a PhysicalExecutionPlan,
        identification: IdentificationResult,
        estimand: IdentifiedEstimand,
        indexer: &TemporalIndexer,
        query: &TemporalEffectQuery,
        meta: FitFinishMeta,
    ) -> IdentifiedExecuteFinish<'a> {
        let FitFinishMeta { identify_cached, structure_version, wall_time_ns } = meta;
        let (estimate_artifact, estimate_op) = self.estimate_provenance;
        let certificate = crate::Identification::Point {
            result: identification.clone(),
            temporal_indexer: Some(indexer.clone()),
            strategy: IdentifierId::TemporalBackdoorUnfolded,
            structure_version,
        };
        IdentifiedExecuteFinish::new(
            physical,
            identification,
            estimand,
            self.estimate,
            Strategy::new(IdentifierId::TemporalBackdoorUnfolded, self.estimator_id),
            query.treatment,
            query.outcome,
        )
        .identify_cached(identify_cached)
        .refutations(self.refutations)
        .wall_time_ns(wall_time_ns)
        .bootstrap_replicates_ok(self.bootstrap_replicates_ok)
        .cancelled(self.cancelled)
        .extras(IdentifiedExecuteExtras {
            certificate: Some(certificate),
            identify_provenance: Some(provenance_ids(
                "identify.temporal_backdoor",
                "identify.temporal.backdoor.unfolded",
            )),
            estimate_provenance: Some(provenance_ids(estimate_artifact, estimate_op)),
            posterior: self.posterior,
            diagnostics: Some(self.diagnostics),
            predictive_checks: self.predictive_checks,
            ..Default::default()
        })
    }
}

/// Strip the interval and posterior a point-only fit never publishes.
pub(super) fn seal_point_only(point_only: bool, mut result: StudyResult) -> StudyResult {
    if point_only {
        result.interval = None;
        result.posterior = None;
    }
    result
}

const SEQUENTIAL_PROVENANCE: (&str, &str) =
    ("estimate.temporal_sequential_gcomp", "estimate.temporal.sequential.gcomp");
const LINEAR_PROVENANCE: (&str, &str) =
    ("estimate.temporal_linear_adjustment", "estimate.temporal.linear.adjustment");
const BAYESIAN_PROVENANCE: (&str, &str) =
    ("estimate.bayesian_temporal_gcomp", "estimate.bayesian.temporal.gcomp");

/// Whether `config` carries a coefficient prior the isotropic temporal fits
/// cannot take.
fn has_mapped_prior(config: &BayesianConfig) -> bool {
    config.has_transfer()
}

/// Fit and assemble the temporal effect of `inputs`.
pub(super) fn fit_temporal_effect(
    inputs: &TemporalEffectInputs<'_>,
    ctx: &ExecutionContext,
) -> Result<TemporalEffectFit, CausalError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_ESTIMATE_POINT });
    }
    require_identified(inputs.identification)?;
    if inputs.query.is_multi_step_sustained() {
        fit_sustained(inputs, ctx)
    } else if !inputs.query.effect_modifiers.is_empty() {
        fit_conditional(inputs, ctx)
    } else {
        fit_single_window(inputs, ctx)
    }
}

fn fit_sustained(
    inputs: &TemporalEffectInputs<'_>,
    ctx: &ExecutionContext,
) -> Result<TemporalEffectFit, CausalError> {
    let TemporalEffectInputs { data, graph, query, identification, estimand, indexer, .. } =
        *inputs;
    if inputs.split.is_some() {
        return Err(CausalError::Unsupported {
            message: "multi-step sustained g-computation currently requires no discovery-estimation split",
        });
    }
    let bayes = if let InferenceMode::Bayesian(cfg) = inputs.inference {
        if has_mapped_prior(cfg) {
            return Err(CausalError::Unsupported {
                message: "multi-step sustained inference requires isotropic per-mechanism priors",
            });
        }
        Some(bayesian_gcomp(cfg, ctx))
    } else {
        None
    };
    let mut assumptions = identification.required_assumptions.clone();
    assumptions.push(sequential_linear_sem_assumption(SUSTAINED_LINEAR_SEM));
    let mut mechanisms = Vec::new();
    let (estimate, mut posterior) =
        antecedent_estimate::temporal_sequential::estimate_sustained_window_with_validation(
            &antecedent_estimate::SequentialProblem { data, graph, indexer, estimand },
            query,
            antecedent_estimate::SequentialInference {
                status: identification.status,
                assumptions,
                bootstrap_replicates: inputs.bootstrap_replicates,
                bayesian: bayes.as_ref(),
            },
            ctx,
            Some(&mut mechanisms),
        )
        .map_err(CausalError::from)?;
    let atom = super::sequential_validation::SequentialValidationAtom {
        weight: 1.0,
        graph: graph.clone(),
        indexer: indexer.clone(),
        estimand: estimand.clone(),
        status: identification.status,
        estimate: estimate.clone(),
        mechanisms,
    };
    let (refutations, mut diagnostics, predictive_checks) =
        super::sequential_validation::validate_sequential(
            SequentialValidation {
                data,
                query,
                suite: inputs.refute,
                custom: inputs.custom_validators,
                predictive_sims: crate::analysis::latency::predictive_check_sims(
                    inputs.latency_mode,
                ),
            },
            &[atom],
            bayes.as_ref(),
            posterior.as_mut(),
            estimate.ate,
            ctx,
        )?;
    // The validation diagnostics are the prebuilt list, so the window note
    // and the dependence caveats join them.
    diagnostics.push(Diagnostic::new(
        "estimate.temporal.sustained_window",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "the contrast propagates through all intervened times in topological order; no \
         one-node collapse",
    ));
    let estimate = if bayes.is_none() {
        diagnostics.extend(super::temporal_path::sequential_dependence_se_diagnostics(
            &estimate,
            inputs.bootstrap_replicates,
        ));
        estimate.with_block_family(antecedent_estimate::CircularBlockFamily::Sequential)
    } else {
        estimate
    };
    Ok(TemporalEffectFit {
        bootstrap_replicates_ok: estimate.uncertainty.bootstrap.replicates_ok,
        cancelled: estimate.uncertainty.bootstrap.cancelled,
        estimate,
        estimator_id: EstimatorId::TemporalSequentialGcomp,
        refutations,
        diagnostics,
        posterior,
        predictive_checks,
        point_only: false,
        estimate_provenance: SEQUENTIAL_PROVENANCE,
    })
}

fn fit_conditional(
    inputs: &TemporalEffectInputs<'_>,
    ctx: &ExecutionContext,
) -> Result<TemporalEffectFit, CausalError> {
    let query = inputs.query;
    if query.effect_modifiers.len() != 1
        || matches!(query.policy, antecedent_core::TemporalPolicy::Dynamic { .. })
    {
        return Err(CausalError::Unsupported {
            message: "temporal conditional effect fits one modifier on a single-step pulse or sustained policy",
        });
    }
    if inputs.refute != RefuteSuite::None {
        return Err(CausalError::Unsupported {
            message: "temporal conditional effect is point-only; cheap/full refuters are not licensed",
        });
    }
    let prior_scale = match inputs.inference {
        InferenceMode::Bayesian(cfg) => {
            if has_mapped_prior(cfg) {
                return Err(CausalError::Unsupported {
                    message: "temporal conditional effect uses the isotropic conjugate prior scale; a mapped coefficient prior is not licensed",
                });
            }
            Some(cfg.prior_scale)
        }
        InferenceMode::Frequentist => None,
    };
    let point = antecedent_estimate::temporal_conditional_marginal_effect(
        inputs.data,
        inputs.estimand,
        query,
        inputs.indexer,
        &ctx.kernel_policy,
        prior_scale,
    )?;
    let estimate = EffectEstimate::new(
        point,
        f64::NAN,
        inputs.identification.required_assumptions.clone(),
        OverlapPolicy::ExplicitOverride,
    );
    Ok(TemporalEffectFit {
        estimate,
        estimator_id: EstimatorId::TemporalLinearAdjustment,
        refutations: Vec::new(),
        diagnostics: vec![Diagnostic::new(
            "estimate.temporal.conditional_marginal",
            DiagnosticKind::Execution,
            DiagnosticSeverity::Info,
            "published value is the one-modifier marginal effect on the unfolded temporal design; no interval is published",
        )],
        posterior: None,
        predictive_checks: Vec::new(),
        bootstrap_replicates_ok: None,
        cancelled: false,
        point_only: true,
        estimate_provenance: LINEAR_PROVENANCE,
    })
}

/// The Bayesian temporal g-computation of a single window: the sealed
/// operation's fit when one was admitted, otherwise a fresh fit under the
/// resolved prior; returns the estimator and design every check reuses.
fn fit_single_window_bayesian(
    inputs: &TemporalEffectInputs<'_>,
    cfg: &crate::BayesianConfig,
    prep: &antecedent_estimate::PreparedEstimationProblem,
    ctx: &ExecutionContext,
) -> Result<
    (EffectEstimate, CausalPosterior, (BayesianTemporalGcomp, PreparedBayesianProblem)),
    CausalError,
> {
    let TemporalEffectInputs {
        data,
        query,
        identification,
        estimand,
        indexer,
        checked_bayesian,
        ..
    } = *inputs;
    if let Some(operation) = checked_bayesian {
        let fit = crate::analysis::checked_bayesian_temporal_effect::fit_temporal_dag_effect(
            operation,
            data,
            inputs.split,
            ctx,
        )?;
        return Ok((fit.estimate, fit.posterior, (fit.estimator, fit.prepared)));
    }
    let mut bayes = bayesian_temporal_gcomp(cfg, ctx);
    let names = antecedent_estimate::temporal_coefficient_names(data, estimand, query, indexer)
        .map_err(CausalError::from)?;
    let bprep =
        BayesianGComputationAte::from_prepared_temporal(prep, names).map_err(CausalError::from)?;
    let (resolved_prior, conflict_summary) =
        resolve_bayesian_prior_with_conflict(cfg, &bprep, Some(ctx))?;
    bayes.inner.prior = resolved_prior;
    let mut ws = BayesianGCompWorkspace::default();
    let mut posterior =
        bayes.fit(&bprep, identification.status, &mut ws, ctx).map_err(CausalError::from)?;
    if let Some(summary) = conflict_summary {
        posterior = with_conflict_summary(posterior, summary);
    }
    let estimate = effect_from_posterior(&posterior)?;
    Ok((estimate, posterior, (bayes, bprep)))
}

/// The prior/posterior predictive checks of a Bayesian single window (with
/// the lag-1 residual discrepancy and prior sensitivity on `Full`), appended
/// to `refutations`; returns the checks and the posterior carrying any prior
/// sensitivity summary.
fn single_window_predictive_checks(
    inputs: &TemporalEffectInputs<'_>,
    (bayes, bprep): (&BayesianTemporalGcomp, &PreparedBayesianProblem),
    post: &CausalPosterior,
    estimate: &EffectEstimate,
    refutations: &mut Vec<antecedent_validate::RefutationReport>,
    ctx: &ExecutionContext,
) -> Result<(Vec<PredictiveCheckReport>, Option<CausalPosterior>), CausalError> {
    const PPC_ALPHA: f64 = 0.05;
    let refute = inputs.refute;
    let mut predictive_checks = Vec::new();
    let mut posterior = None;
    let sims = crate::analysis::latency::predictive_check_sims(inputs.latency_mode);
    let prior_rep = PriorPredictiveCheck::for_estimator(&bayes.inner, ctx)
        .with_n_sims(sims)
        .check_with_prior(bprep, &bayes.inner.prior_in_force(bprep.design.ncols), ctx)
        .map_err(CausalError::from)?;
    refutations.push(prior_rep.to_refutation_report(estimate.ate, PPC_ALPHA));
    predictive_checks.push(prior_rep);

    // `full` adds the lag-1 residual-autocorrelation discrepancy.
    let post_check = PosteriorPredictiveCheck::for_estimator(&bayes.inner, ctx).with_n_sims(sims);
    let post_rep = if matches!(refute, RefuteSuite::Full) {
        post_check.check_temporal(bprep, post, ctx.rng.master_seed())
    } else {
        post_check.check(bprep, post)
    }
    .map_err(CausalError::from)?;
    refutations.push(post_rep.to_refutation_report(estimate.ate, PPC_ALPHA));
    predictive_checks.push(post_rep);

    if matches!(refute, RefuteSuite::Full) {
        let InferenceMode::Bayesian(cfg) = inputs.inference else {
            return Err(CausalError::Unsupported {
                message: "Bayesian predictive checks require inference=Bayesian",
            });
        };
        let mut ws = BayesianGCompWorkspace::default();
        let (summary, sens) = evaluate_bayesian_prior_sensitivity(
            cfg,
            &bayes.inner,
            bprep,
            inputs.identification.status,
            post,
            &mut ws,
            ctx,
        )?;
        refutations.push(sens.to_report(&summary, estimate.ate));
        posterior = Some(with_prior_sensitivity(post.clone(), summary));
    }
    Ok((predictive_checks, posterior))
}

/// The refutation suite of a single-window effect with its dependence-honest
/// SE and PAG-completion notes.
fn single_window_refutations(
    inputs: &TemporalEffectInputs<'_>,
    estimate: &EffectEstimate,
    estimator_id: EstimatorId,
    dependence_se: Option<&antecedent_estimate::TemporalDependenceSe>,
    ctx: &ExecutionContext,
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
    let TemporalEffectInputs { data, query, estimand, indexer, refute, .. } = *inputs;
    let tabular = TabularData::new(data.storage().clone());
    let average = temporal_refuter_average(query);
    let mut workspace = EstimationWorkspace::default();
    let temporal_ctx = TemporalRefitContext {
        indexer,
        temporal_query: query,
        split: inputs.split,
        kernel_policy: &ctx.kernel_policy,
        time_index: Some(data.time_index()),
        panel: None,
    };
    let (refutations, mut diagnostics) = RefutationRun {
        data: &tabular,
        estimand,
        query: &average,
        estimate,
        suite: refute,
        estimator: estimator_id.as_str(),
        custom: inputs.custom_validators,
        temporal: Some(temporal_ctx),
    }
    .reports(&mut workspace, None, ctx)?;
    if let Some(info) = dependence_se {
        diagnostics
            .extend(super::temporal_path::single_window_dependence_se_diagnostics(estimate, info));
    }
    if inputs.completed_to_dag {
        diagnostics.push(Diagnostic::new(
            "temporal.pag.completed_to_dag",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "TemporalPag completed to TemporalDag before temporal.backdoor \
             (completion path; not class-aware temporal PAG identification)",
        ));
    }

    Ok((refutations, diagnostics))
}

fn fit_single_window(
    inputs: &TemporalEffectInputs<'_>,
    ctx: &ExecutionContext,
) -> Result<TemporalEffectFit, CausalError> {
    let TemporalEffectInputs { data, query, identification, estimand, indexer, refute, .. } =
        *inputs;
    let mut estimator = TemporalLinearAdjustment::new();
    estimator.inner.bootstrap_replicates = inputs.bootstrap_replicates;
    estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
    let prep = estimator
        .prepare(data, estimand, query, indexer, inputs.split, &ctx.kernel_policy)
        .map_err(CausalError::from)?;

    // Bayesian fits keep the estimator (with the resolved prior in force) and the
    // lag-named, serial-dependence-tempered design so every check reuses both.
    let mut bayes_fit: Option<(BayesianTemporalGcomp, PreparedBayesianProblem)> = None;
    let mut dependence_se = None;
    let (estimate, posterior, estimate_provenance) = match inputs.inference {
        InferenceMode::Bayesian(cfg) => {
            let (estimate, posterior, fit) = fit_single_window_bayesian(inputs, cfg, &prep, ctx)?;
            bayes_fit = Some(fit);
            (estimate, Some(posterior), BAYESIAN_PROVENANCE)
        }
        InferenceMode::Frequentist => {
            // One series: lagged rows are serially dependent, so the only SE is a
            // fixed-b-scaled circular-block bootstrap over consecutive lag-aligned
            // rows, never the iid OLS / iid row-bootstrap pair of the static estimator.
            let (estimate, info) = estimator
                .fit_dependence_honest(
                    &prep,
                    indexer,
                    ctx,
                    identification.required_assumptions.clone(),
                )
                .map_err(CausalError::from)?;
            dependence_se = Some(info);
            (
                estimate.with_block_family(antecedent_estimate::CircularBlockFamily::SingleWindow),
                None,
                LINEAR_PROVENANCE,
            )
        }
    };
    let estimate = estimate.with_n_obs(u64::try_from(prep.design.nrows).unwrap_or(u64::MAX));

    let estimator_id = if posterior.is_some() {
        EstimatorId::BayesianTemporalGcomp
    } else {
        EstimatorId::TemporalLinearAdjustment
    };
    let (mut refutations, mut diagnostics) =
        single_window_refutations(inputs, &estimate, estimator_id, dependence_se.as_ref(), ctx)?;

    // Bayesian temporal: prior/posterior PPC + prior sensitivity on Full (mirror static).
    // Both run against the resolved prior in force and the fitted (tempered) design.
    let mut posterior = posterior;
    let mut predictive_checks = Vec::new();
    if !matches!(refute, RefuteSuite::None) {
        if let (Some(fit), Some(post)) = (bayes_fit.as_ref(), posterior.as_ref()) {
            let (checks, sensitive) = single_window_predictive_checks(
                inputs,
                (&fit.0, &fit.1),
                post,
                &estimate,
                &mut refutations,
                ctx,
            )?;
            predictive_checks = checks;
            if sensitive.is_some() {
                posterior = sensitive;
            }
        }
    }
    if let Some(summary) = posterior.as_ref().and_then(|p| p.conflict_summary.as_ref()) {
        push_conflict_diagnostics(&mut diagnostics, summary);
    }
    Ok(TemporalEffectFit {
        bootstrap_replicates_ok: estimate.uncertainty.bootstrap.replicates_ok,
        cancelled: estimate.uncertainty.bootstrap.cancelled,
        estimate,
        estimator_id,
        refutations,
        diagnostics,
        posterior,
        predictive_checks,
        point_only: false,
        estimate_provenance,
    })
}

/// Whether the plan's discovery algorithm completed a PAG to a DAG.
pub(super) fn plan_completed_to_dag(physical: &PhysicalExecutionPlan) -> bool {
    physical
        .logical
        .record
        .discovery_algorithm
        .as_deref()
        .is_some_and(|a| a.contains("pag_completed_to_dag") || a.contains("completed_to_dag"))
}
